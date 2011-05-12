#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <signal.h>
#include <fcntl.h>
#include <netinet/in.h>

#include "debug/Debugger.h"
#include "log/Logger.h"
#include "steering/ClientConnection.h"
#include "HttpPost.h"

namespace hemelb
{
  namespace steering
  {
    ClientConnection::ClientConnection(int iSteeringSessionId)
    {
      sem_init(&mIsBusy, 0, 1);

      // Write the name of this machine to a file.

      {
        char thisMachineName[255];
        gethostname(thisMachineName, 255);
        FILE *f = fopen("env_details.asc", "w");
        fprintf(f, "%s\n", thisMachineName);
        fclose(f);
      }

      // Send the steering session id we're using to the rendezvous resource.
      {
        char steering_session_id_char[255];
        std::sprintf(steering_session_id_char, "%i", iSteeringSessionId);

        HttpPost::request("bunsen.chem.ucl.ac.uk",
                          28080,
                          "/ahe/test/rendezvous/",
                          steering_session_id_char);
      }

      mCurrentSocket = -1;
      mIsBroken = false;

      // Create the socket.
      mListeningSocket = socket(AF_INET, SOCK_STREAM, 0);
      if (mListeningSocket == -1)
      {
        perror("socket");
        exit(1);
      }

      // Make the socket reusable.
      int yes = 1;
      if (setsockopt(mListeningSocket, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(int)) == -1)
      {
        perror("setsockopt");
        exit(1);
      }

      // Bind to the socket.
      {
        struct sockaddr_in my_address;

        my_address.sin_family = AF_INET;

        debug::Debugger::Get()->BreakHere();
        my_address.sin_port = htons((in_port_t) MYPORT);
        my_address.sin_addr.s_addr = INADDR_ANY;
        memset(my_address.sin_zero, '\0', sizeof my_address.sin_zero);

        if (bind(mListeningSocket, (struct sockaddr *) &my_address, sizeof my_address) == -1)
        {
          perror("bind");
          exit(1);
        }
      }

      // Mark the socket as accepting incoming connections.
      if (listen(mListeningSocket, CONNECTION_BACKLOG) == -1)
      {
        perror("listen");
        exit(1);
      }
    }

    ClientConnection::~ClientConnection()
    {
      sem_destroy(&mIsBusy);

      close(mCurrentSocket);
      close(mListeningSocket);
    }

    int ClientConnection::GetWorkingSocket()
    {
      sem_wait(&mIsBusy);

      // If we haven't yet had a socket, or the current one is broken, open a new one.
      if (mCurrentSocket < 0 || mIsBroken)
      {
        // Accept an incoming connection from the client.
        struct sockaddr_in clientAddress;
        socklen_t socketSize = sizeof (clientAddress);

        int lOldSocket = mCurrentSocket;

        // Make the socket non-blocking, just while we try to accept on it, then set
        // it back to what it was before.
        {
          int flags = fcntl(mListeningSocket, F_GETFL, 0);
          if (flags == -1)
          {
            flags = 0;
          }
          if (fcntl(mListeningSocket, F_SETFL, flags | O_NONBLOCK) < 0)
          {
            perror("flags");
          }

          // Try to accept a socket (from the non-blocking socket)
          mCurrentSocket
              = accept(mListeningSocket, (struct sockaddr *) &clientAddress, &socketSize);

          // We've got a socket - make that socket non-blocking too.
          if (mCurrentSocket > 0)
          {
            flags = fcntl(mCurrentSocket, F_GETFL, 0);
            if (flags == -1)
            {
              flags = 0;
            }
            if (fcntl(mCurrentSocket, F_SETFL, flags | O_NONBLOCK) < 0)
            {
              perror("flags");
            }
          }
        }

        // If we had a socket before, close it.
        if (lOldSocket > 0)
        {
          close(lOldSocket);
        }

        // We've only just created the socket so it shouldn't be broken.
        mIsBroken = false;
      }

      int lRet = mCurrentSocket;

      sem_post(&mIsBusy);

      return lRet;
    }

    void ClientConnection::ReportBroken(int iSocketNum)
    {
      sem_wait(&mIsBusy);
      if (mCurrentSocket == iSocketNum)
      {
        mIsBroken = true;
      }
      sem_post(&mIsBusy);
    }

  }
}

